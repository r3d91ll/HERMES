#!/bin/bash
# Configure ArangoDB to listen on network interface and setup firewall rules

echo "Configuring ArangoDB for network access..."

# Backup the original config
if [ ! -f /etc/arangodb3/arangod.conf.backup ]; then
    cp /etc/arangodb3/arangod.conf /etc/arangodb3/arangod.conf.backup
    echo "✓ Created backup of arangod.conf"
fi

# Update ArangoDB to listen on all interfaces (we'll restrict with firewall)
sed -i 's/endpoint = tcp:\/\/127.0.0.1:8529/endpoint = tcp:\/\/0.0.0.0:8529/' /etc/arangodb3/arangod.conf
echo "✓ Updated ArangoDB to listen on all interfaces"

# Setup firewall rules
echo "Setting up firewall rules..."

# Check if firewalld is running
if systemctl is-active --quiet firewalld; then
    echo "Using firewalld..."
    
    # Create a new zone for the lab network if it doesn't exist
    firewall-cmd --permanent --new-zone=lab-network 2>/dev/null || echo "Zone already exists"
    
    # Add the lab network to the zone
    firewall-cmd --permanent --zone=lab-network --add-source=192.168.1.0/24
    
    # Add the required services/ports to the zone
    firewall-cmd --permanent --zone=lab-network --add-port=8529/tcp  # ArangoDB
    firewall-cmd --permanent --zone=lab-network --add-port=80/tcp    # HTTP
    firewall-cmd --permanent --zone=lab-network --add-port=443/tcp   # HTTPS
    firewall-cmd --permanent --zone=lab-network --add-port=22/tcp    # SSH
    
    # Reload firewall
    firewall-cmd --reload
    
    echo "✓ Firewall rules configured"
    echo "  - Created lab-network zone"
    echo "  - Allowed access from 192.168.1.0/24"
    echo "  - Opened ports: 8529, 80, 443, 22"
else
    echo "firewalld is not running. Checking for ufw..."
    
    if command -v ufw &> /dev/null; then
        # Using ufw
        ufw allow from 192.168.1.0/24 to any port 8529 comment "ArangoDB"
        ufw allow from 192.168.1.0/24 to any port 80 comment "HTTP"
        ufw allow from 192.168.1.0/24 to any port 443 comment "HTTPS"
        ufw allow from 192.168.1.0/24 to any port 22 comment "SSH"
        echo "✓ UFW rules configured"
    else
        echo "⚠️  No firewall detected. Please configure manually."
    fi
fi

# Restart ArangoDB
echo "Restarting ArangoDB service..."
systemctl restart arangodb3

# Wait for service to come up
sleep 5

# Check service status
if systemctl is-active --quiet arangodb3; then
    echo "✓ ArangoDB service is running"
    
    # Show the endpoint
    echo ""
    echo "ArangoDB is now accessible at:"
    echo "  - http://192.168.1.69:8529"
    echo "  - From any host in 192.168.1.0/24 network"
else
    echo "❌ ArangoDB service failed to start"
    echo "Check logs: journalctl -u arangodb3 -n 50"
fi

echo ""
echo "Done! Next steps:"
echo "1. Update your .env file to use ARANGO_HOST=192.168.1.69"
echo "2. Test connection: curl http://192.168.1.69:8529/_api/version"